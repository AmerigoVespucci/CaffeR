// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: GenSeed.proto

#define INTERNAL_SUPPRESS_PROTOBUF_FIELD_DEPRECATION
#include "GenSeed.pb.h"

#include <algorithm>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/once.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/wire_format_lite_inl.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)

namespace {

const ::google::protobuf::Descriptor* CaffeGenSeed_descriptor_ = NULL;
const ::google::protobuf::internal::GeneratedMessageReflection*
  CaffeGenSeed_reflection_ = NULL;
const ::google::protobuf::EnumDescriptor* CaffeGenSeed_NetEndType_descriptor_ = NULL;

}  // namespace


void protobuf_AssignDesc_GenSeed_2eproto() {
  protobuf_AddDesc_GenSeed_2eproto();
  const ::google::protobuf::FileDescriptor* file =
    ::google::protobuf::DescriptorPool::generated_pool()->FindFileByName(
      "GenSeed.proto");
  GOOGLE_CHECK(file != NULL);
  CaffeGenSeed_descriptor_ = file->message_type(0);
  static const int CaffeGenSeed_offsets_[7] = {
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(CaffeGenSeed, test_list_file_name_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(CaffeGenSeed, train_list_file_name_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(CaffeGenSeed, num_test_cases_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(CaffeGenSeed, net_end_type_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(CaffeGenSeed, num_output_nodes_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(CaffeGenSeed, proto_file_name_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(CaffeGenSeed, model_file_name_),
  };
  CaffeGenSeed_reflection_ =
    new ::google::protobuf::internal::GeneratedMessageReflection(
      CaffeGenSeed_descriptor_,
      CaffeGenSeed::default_instance_,
      CaffeGenSeed_offsets_,
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(CaffeGenSeed, _has_bits_[0]),
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(CaffeGenSeed, _unknown_fields_),
      -1,
      ::google::protobuf::DescriptorPool::generated_pool(),
      ::google::protobuf::MessageFactory::generated_factory(),
      sizeof(CaffeGenSeed));
  CaffeGenSeed_NetEndType_descriptor_ = CaffeGenSeed_descriptor_->enum_type(0);
}

namespace {

GOOGLE_PROTOBUF_DECLARE_ONCE(protobuf_AssignDescriptors_once_);
inline void protobuf_AssignDescriptorsOnce() {
  ::google::protobuf::GoogleOnceInit(&protobuf_AssignDescriptors_once_,
                 &protobuf_AssignDesc_GenSeed_2eproto);
}

void protobuf_RegisterTypes(const ::std::string&) {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedMessage(
    CaffeGenSeed_descriptor_, &CaffeGenSeed::default_instance());
}

}  // namespace

void protobuf_ShutdownFile_GenSeed_2eproto() {
  delete CaffeGenSeed::default_instance_;
  delete CaffeGenSeed_reflection_;
}

void protobuf_AddDesc_GenSeed_2eproto() {
  static bool already_here = false;
  if (already_here) return;
  already_here = true;
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  ::google::protobuf::DescriptorPool::InternalAddGeneratedFile(
    "\n\rGenSeed.proto\"\236\002\n\014CaffeGenSeed\022\033\n\023test"
    "_list_file_name\030\001 \002(\t\022\034\n\024train_list_file"
    "_name\030\002 \002(\t\022\026\n\016num_test_cases\030\003 \002(\005\022.\n\014n"
    "et_end_type\030\004 \002(\0162\030.CaffeGenSeed.NetEndT"
    "ype\022\030\n\020num_output_nodes\030\005 \001(\005\022\027\n\017proto_f"
    "ile_name\030\006 \002(\t\022\027\n\017model_file_name\030\007 \002(\t\""
    "\?\n\nNetEndType\022\r\n\tEND_VALID\020\001\022\017\n\013END_ONE_"
    "HOT\020\002\022\021\n\rEND_MULTI_HOT\020\003", 304);
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(
    "GenSeed.proto", &protobuf_RegisterTypes);
  CaffeGenSeed::default_instance_ = new CaffeGenSeed();
  CaffeGenSeed::default_instance_->InitAsDefaultInstance();
  ::google::protobuf::internal::OnShutdown(&protobuf_ShutdownFile_GenSeed_2eproto);
}

// Force AddDescriptors() to be called at static initialization time.
struct StaticDescriptorInitializer_GenSeed_2eproto {
  StaticDescriptorInitializer_GenSeed_2eproto() {
    protobuf_AddDesc_GenSeed_2eproto();
  }
} static_descriptor_initializer_GenSeed_2eproto_;

// ===================================================================

const ::google::protobuf::EnumDescriptor* CaffeGenSeed_NetEndType_descriptor() {
  protobuf_AssignDescriptorsOnce();
  return CaffeGenSeed_NetEndType_descriptor_;
}
bool CaffeGenSeed_NetEndType_IsValid(int value) {
  switch(value) {
    case 1:
    case 2:
    case 3:
      return true;
    default:
      return false;
  }
}

#ifndef _MSC_VER
const CaffeGenSeed_NetEndType CaffeGenSeed::END_VALID;
const CaffeGenSeed_NetEndType CaffeGenSeed::END_ONE_HOT;
const CaffeGenSeed_NetEndType CaffeGenSeed::END_MULTI_HOT;
const CaffeGenSeed_NetEndType CaffeGenSeed::NetEndType_MIN;
const CaffeGenSeed_NetEndType CaffeGenSeed::NetEndType_MAX;
const int CaffeGenSeed::NetEndType_ARRAYSIZE;
#endif  // _MSC_VER
#ifndef _MSC_VER
const int CaffeGenSeed::kTestListFileNameFieldNumber;
const int CaffeGenSeed::kTrainListFileNameFieldNumber;
const int CaffeGenSeed::kNumTestCasesFieldNumber;
const int CaffeGenSeed::kNetEndTypeFieldNumber;
const int CaffeGenSeed::kNumOutputNodesFieldNumber;
const int CaffeGenSeed::kProtoFileNameFieldNumber;
const int CaffeGenSeed::kModelFileNameFieldNumber;
#endif  // !_MSC_VER

CaffeGenSeed::CaffeGenSeed()
  : ::google::protobuf::Message() {
  SharedCtor();
}

void CaffeGenSeed::InitAsDefaultInstance() {
}

CaffeGenSeed::CaffeGenSeed(const CaffeGenSeed& from)
  : ::google::protobuf::Message() {
  SharedCtor();
  MergeFrom(from);
}

void CaffeGenSeed::SharedCtor() {
  _cached_size_ = 0;
  test_list_file_name_ = const_cast< ::std::string*>(&::google::protobuf::internal::kEmptyString);
  train_list_file_name_ = const_cast< ::std::string*>(&::google::protobuf::internal::kEmptyString);
  num_test_cases_ = 0;
  net_end_type_ = 1;
  num_output_nodes_ = 0;
  proto_file_name_ = const_cast< ::std::string*>(&::google::protobuf::internal::kEmptyString);
  model_file_name_ = const_cast< ::std::string*>(&::google::protobuf::internal::kEmptyString);
  ::memset(_has_bits_, 0, sizeof(_has_bits_));
}

CaffeGenSeed::~CaffeGenSeed() {
  SharedDtor();
}

void CaffeGenSeed::SharedDtor() {
  if (test_list_file_name_ != &::google::protobuf::internal::kEmptyString) {
    delete test_list_file_name_;
  }
  if (train_list_file_name_ != &::google::protobuf::internal::kEmptyString) {
    delete train_list_file_name_;
  }
  if (proto_file_name_ != &::google::protobuf::internal::kEmptyString) {
    delete proto_file_name_;
  }
  if (model_file_name_ != &::google::protobuf::internal::kEmptyString) {
    delete model_file_name_;
  }
  if (this != default_instance_) {
  }
}

void CaffeGenSeed::SetCachedSize(int size) const {
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
}
const ::google::protobuf::Descriptor* CaffeGenSeed::descriptor() {
  protobuf_AssignDescriptorsOnce();
  return CaffeGenSeed_descriptor_;
}

const CaffeGenSeed& CaffeGenSeed::default_instance() {
  if (default_instance_ == NULL) protobuf_AddDesc_GenSeed_2eproto();
  return *default_instance_;
}

CaffeGenSeed* CaffeGenSeed::default_instance_ = NULL;

CaffeGenSeed* CaffeGenSeed::New() const {
  return new CaffeGenSeed;
}

void CaffeGenSeed::Clear() {
  if (_has_bits_[0 / 32] & (0xffu << (0 % 32))) {
    if (has_test_list_file_name()) {
      if (test_list_file_name_ != &::google::protobuf::internal::kEmptyString) {
        test_list_file_name_->clear();
      }
    }
    if (has_train_list_file_name()) {
      if (train_list_file_name_ != &::google::protobuf::internal::kEmptyString) {
        train_list_file_name_->clear();
      }
    }
    num_test_cases_ = 0;
    net_end_type_ = 1;
    num_output_nodes_ = 0;
    if (has_proto_file_name()) {
      if (proto_file_name_ != &::google::protobuf::internal::kEmptyString) {
        proto_file_name_->clear();
      }
    }
    if (has_model_file_name()) {
      if (model_file_name_ != &::google::protobuf::internal::kEmptyString) {
        model_file_name_->clear();
      }
    }
  }
  ::memset(_has_bits_, 0, sizeof(_has_bits_));
  mutable_unknown_fields()->Clear();
}

bool CaffeGenSeed::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!(EXPRESSION)) return false
  ::google::protobuf::uint32 tag;
  while ((tag = input->ReadTag()) != 0) {
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // required string test_list_file_name = 1;
      case 1: {
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadString(
                input, this->mutable_test_list_file_name()));
          ::google::protobuf::internal::WireFormat::VerifyUTF8String(
            this->test_list_file_name().data(), this->test_list_file_name().length(),
            ::google::protobuf::internal::WireFormat::PARSE);
        } else {
          goto handle_uninterpreted;
        }
        if (input->ExpectTag(18)) goto parse_train_list_file_name;
        break;
      }

      // required string train_list_file_name = 2;
      case 2: {
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED) {
         parse_train_list_file_name:
          DO_(::google::protobuf::internal::WireFormatLite::ReadString(
                input, this->mutable_train_list_file_name()));
          ::google::protobuf::internal::WireFormat::VerifyUTF8String(
            this->train_list_file_name().data(), this->train_list_file_name().length(),
            ::google::protobuf::internal::WireFormat::PARSE);
        } else {
          goto handle_uninterpreted;
        }
        if (input->ExpectTag(24)) goto parse_num_test_cases;
        break;
      }

      // required int32 num_test_cases = 3;
      case 3: {
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_VARINT) {
         parse_num_test_cases:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::int32, ::google::protobuf::internal::WireFormatLite::TYPE_INT32>(
                 input, &num_test_cases_)));
          set_has_num_test_cases();
        } else {
          goto handle_uninterpreted;
        }
        if (input->ExpectTag(32)) goto parse_net_end_type;
        break;
      }

      // required .CaffeGenSeed.NetEndType net_end_type = 4;
      case 4: {
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_VARINT) {
         parse_net_end_type:
          int value;
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   int, ::google::protobuf::internal::WireFormatLite::TYPE_ENUM>(
                 input, &value)));
          if (::CaffeGenSeed_NetEndType_IsValid(value)) {
            set_net_end_type(static_cast< ::CaffeGenSeed_NetEndType >(value));
          } else {
            mutable_unknown_fields()->AddVarint(4, value);
          }
        } else {
          goto handle_uninterpreted;
        }
        if (input->ExpectTag(40)) goto parse_num_output_nodes;
        break;
      }

      // optional int32 num_output_nodes = 5;
      case 5: {
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_VARINT) {
         parse_num_output_nodes:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::int32, ::google::protobuf::internal::WireFormatLite::TYPE_INT32>(
                 input, &num_output_nodes_)));
          set_has_num_output_nodes();
        } else {
          goto handle_uninterpreted;
        }
        if (input->ExpectTag(50)) goto parse_proto_file_name;
        break;
      }

      // required string proto_file_name = 6;
      case 6: {
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED) {
         parse_proto_file_name:
          DO_(::google::protobuf::internal::WireFormatLite::ReadString(
                input, this->mutable_proto_file_name()));
          ::google::protobuf::internal::WireFormat::VerifyUTF8String(
            this->proto_file_name().data(), this->proto_file_name().length(),
            ::google::protobuf::internal::WireFormat::PARSE);
        } else {
          goto handle_uninterpreted;
        }
        if (input->ExpectTag(58)) goto parse_model_file_name;
        break;
      }

      // required string model_file_name = 7;
      case 7: {
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED) {
         parse_model_file_name:
          DO_(::google::protobuf::internal::WireFormatLite::ReadString(
                input, this->mutable_model_file_name()));
          ::google::protobuf::internal::WireFormat::VerifyUTF8String(
            this->model_file_name().data(), this->model_file_name().length(),
            ::google::protobuf::internal::WireFormat::PARSE);
        } else {
          goto handle_uninterpreted;
        }
        if (input->ExpectAtEnd()) return true;
        break;
      }

      default: {
      handle_uninterpreted:
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_END_GROUP) {
          return true;
        }
        DO_(::google::protobuf::internal::WireFormat::SkipField(
              input, tag, mutable_unknown_fields()));
        break;
      }
    }
  }
  return true;
#undef DO_
}

void CaffeGenSeed::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // required string test_list_file_name = 1;
  if (has_test_list_file_name()) {
    ::google::protobuf::internal::WireFormat::VerifyUTF8String(
      this->test_list_file_name().data(), this->test_list_file_name().length(),
      ::google::protobuf::internal::WireFormat::SERIALIZE);
    ::google::protobuf::internal::WireFormatLite::WriteString(
      1, this->test_list_file_name(), output);
  }

  // required string train_list_file_name = 2;
  if (has_train_list_file_name()) {
    ::google::protobuf::internal::WireFormat::VerifyUTF8String(
      this->train_list_file_name().data(), this->train_list_file_name().length(),
      ::google::protobuf::internal::WireFormat::SERIALIZE);
    ::google::protobuf::internal::WireFormatLite::WriteString(
      2, this->train_list_file_name(), output);
  }

  // required int32 num_test_cases = 3;
  if (has_num_test_cases()) {
    ::google::protobuf::internal::WireFormatLite::WriteInt32(3, this->num_test_cases(), output);
  }

  // required .CaffeGenSeed.NetEndType net_end_type = 4;
  if (has_net_end_type()) {
    ::google::protobuf::internal::WireFormatLite::WriteEnum(
      4, this->net_end_type(), output);
  }

  // optional int32 num_output_nodes = 5;
  if (has_num_output_nodes()) {
    ::google::protobuf::internal::WireFormatLite::WriteInt32(5, this->num_output_nodes(), output);
  }

  // required string proto_file_name = 6;
  if (has_proto_file_name()) {
    ::google::protobuf::internal::WireFormat::VerifyUTF8String(
      this->proto_file_name().data(), this->proto_file_name().length(),
      ::google::protobuf::internal::WireFormat::SERIALIZE);
    ::google::protobuf::internal::WireFormatLite::WriteString(
      6, this->proto_file_name(), output);
  }

  // required string model_file_name = 7;
  if (has_model_file_name()) {
    ::google::protobuf::internal::WireFormat::VerifyUTF8String(
      this->model_file_name().data(), this->model_file_name().length(),
      ::google::protobuf::internal::WireFormat::SERIALIZE);
    ::google::protobuf::internal::WireFormatLite::WriteString(
      7, this->model_file_name(), output);
  }

  if (!unknown_fields().empty()) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        unknown_fields(), output);
  }
}

::google::protobuf::uint8* CaffeGenSeed::SerializeWithCachedSizesToArray(
    ::google::protobuf::uint8* target) const {
  // required string test_list_file_name = 1;
  if (has_test_list_file_name()) {
    ::google::protobuf::internal::WireFormat::VerifyUTF8String(
      this->test_list_file_name().data(), this->test_list_file_name().length(),
      ::google::protobuf::internal::WireFormat::SERIALIZE);
    target =
      ::google::protobuf::internal::WireFormatLite::WriteStringToArray(
        1, this->test_list_file_name(), target);
  }

  // required string train_list_file_name = 2;
  if (has_train_list_file_name()) {
    ::google::protobuf::internal::WireFormat::VerifyUTF8String(
      this->train_list_file_name().data(), this->train_list_file_name().length(),
      ::google::protobuf::internal::WireFormat::SERIALIZE);
    target =
      ::google::protobuf::internal::WireFormatLite::WriteStringToArray(
        2, this->train_list_file_name(), target);
  }

  // required int32 num_test_cases = 3;
  if (has_num_test_cases()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteInt32ToArray(3, this->num_test_cases(), target);
  }

  // required .CaffeGenSeed.NetEndType net_end_type = 4;
  if (has_net_end_type()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteEnumToArray(
      4, this->net_end_type(), target);
  }

  // optional int32 num_output_nodes = 5;
  if (has_num_output_nodes()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteInt32ToArray(5, this->num_output_nodes(), target);
  }

  // required string proto_file_name = 6;
  if (has_proto_file_name()) {
    ::google::protobuf::internal::WireFormat::VerifyUTF8String(
      this->proto_file_name().data(), this->proto_file_name().length(),
      ::google::protobuf::internal::WireFormat::SERIALIZE);
    target =
      ::google::protobuf::internal::WireFormatLite::WriteStringToArray(
        6, this->proto_file_name(), target);
  }

  // required string model_file_name = 7;
  if (has_model_file_name()) {
    ::google::protobuf::internal::WireFormat::VerifyUTF8String(
      this->model_file_name().data(), this->model_file_name().length(),
      ::google::protobuf::internal::WireFormat::SERIALIZE);
    target =
      ::google::protobuf::internal::WireFormatLite::WriteStringToArray(
        7, this->model_file_name(), target);
  }

  if (!unknown_fields().empty()) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        unknown_fields(), target);
  }
  return target;
}

int CaffeGenSeed::ByteSize() const {
  int total_size = 0;

  if (_has_bits_[0 / 32] & (0xffu << (0 % 32))) {
    // required string test_list_file_name = 1;
    if (has_test_list_file_name()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::StringSize(
          this->test_list_file_name());
    }

    // required string train_list_file_name = 2;
    if (has_train_list_file_name()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::StringSize(
          this->train_list_file_name());
    }

    // required int32 num_test_cases = 3;
    if (has_num_test_cases()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::Int32Size(
          this->num_test_cases());
    }

    // required .CaffeGenSeed.NetEndType net_end_type = 4;
    if (has_net_end_type()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::EnumSize(this->net_end_type());
    }

    // optional int32 num_output_nodes = 5;
    if (has_num_output_nodes()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::Int32Size(
          this->num_output_nodes());
    }

    // required string proto_file_name = 6;
    if (has_proto_file_name()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::StringSize(
          this->proto_file_name());
    }

    // required string model_file_name = 7;
    if (has_model_file_name()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::StringSize(
          this->model_file_name());
    }

  }
  if (!unknown_fields().empty()) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        unknown_fields());
  }
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = total_size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
  return total_size;
}

void CaffeGenSeed::MergeFrom(const ::google::protobuf::Message& from) {
  GOOGLE_CHECK_NE(&from, this);
  const CaffeGenSeed* source =
    ::google::protobuf::internal::dynamic_cast_if_available<const CaffeGenSeed*>(
      &from);
  if (source == NULL) {
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
    MergeFrom(*source);
  }
}

void CaffeGenSeed::MergeFrom(const CaffeGenSeed& from) {
  GOOGLE_CHECK_NE(&from, this);
  if (from._has_bits_[0 / 32] & (0xffu << (0 % 32))) {
    if (from.has_test_list_file_name()) {
      set_test_list_file_name(from.test_list_file_name());
    }
    if (from.has_train_list_file_name()) {
      set_train_list_file_name(from.train_list_file_name());
    }
    if (from.has_num_test_cases()) {
      set_num_test_cases(from.num_test_cases());
    }
    if (from.has_net_end_type()) {
      set_net_end_type(from.net_end_type());
    }
    if (from.has_num_output_nodes()) {
      set_num_output_nodes(from.num_output_nodes());
    }
    if (from.has_proto_file_name()) {
      set_proto_file_name(from.proto_file_name());
    }
    if (from.has_model_file_name()) {
      set_model_file_name(from.model_file_name());
    }
  }
  mutable_unknown_fields()->MergeFrom(from.unknown_fields());
}

void CaffeGenSeed::CopyFrom(const ::google::protobuf::Message& from) {
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void CaffeGenSeed::CopyFrom(const CaffeGenSeed& from) {
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool CaffeGenSeed::IsInitialized() const {
  if ((_has_bits_[0] & 0x0000006f) != 0x0000006f) return false;

  return true;
}

void CaffeGenSeed::Swap(CaffeGenSeed* other) {
  if (other != this) {
    std::swap(test_list_file_name_, other->test_list_file_name_);
    std::swap(train_list_file_name_, other->train_list_file_name_);
    std::swap(num_test_cases_, other->num_test_cases_);
    std::swap(net_end_type_, other->net_end_type_);
    std::swap(num_output_nodes_, other->num_output_nodes_);
    std::swap(proto_file_name_, other->proto_file_name_);
    std::swap(model_file_name_, other->model_file_name_);
    std::swap(_has_bits_[0], other->_has_bits_[0]);
    _unknown_fields_.Swap(&other->_unknown_fields_);
    std::swap(_cached_size_, other->_cached_size_);
  }
}

::google::protobuf::Metadata CaffeGenSeed::GetMetadata() const {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::Metadata metadata;
  metadata.descriptor = CaffeGenSeed_descriptor_;
  metadata.reflection = CaffeGenSeed_reflection_;
  return metadata;
}


// @@protoc_insertion_point(namespace_scope)

// @@protoc_insertion_point(global_scope)
